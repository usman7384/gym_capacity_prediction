import boto3
import pandas as pd
from boto3.dynamodb.conditions import Key

class DynamoDBExporter:
    def __init__(self, table_name, region_name='us-east-1'):
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table = self.dynamodb.Table(table_name)

    def scan_table(self):
        response = self.table.scan()
        data = response['Items']

        # Handle pagination if the dataset is large
        while 'LastEvaluatedKey' in response:
            response = self.table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            data.extend(response['Items'])

        return data

    def save_data(self, data, file_path, format='csv'):
        df = pd.DataFrame(data)

        # Convert types as necessary
        df['occupancy_percentage'] = df['occupancy_percentage'].astype(float)
        df['temperature'] = df['temperature'].astype(float)

        if format == 'csv':
            df.to_csv(file_path, index=False)
        elif format == 'json':
            df.to_json(file_path, orient='records')
        elif format == 'excel':
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

if __name__ == '__main__':
    exporter = DynamoDBExporter(table_name='gym_data',
                                 region_name='eu-central-1')

    data = exporter.scan_table()
    exporter.save_data(data, 'gym_occupancy_data.csv', format='csv')  # Change format as needed
